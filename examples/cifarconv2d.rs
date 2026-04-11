use flate2::read::GzDecoder;
#[allow(unused_imports)]
use image::{Rgb, RgbImage};
use indicatif::ProgressIterator;
use num_traits::cast::ToPrimitive;
use rand::{rng, seq::SliceRandom};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::{
    env,
    fs::File,
    io::{Read, Write},
    path::Path,
    time::{Duration, Instant},
};
use tar::Archive;
#[allow(unused_imports)]
use tenebris::{
    data_augmentation::AugmentationOptions,
    errorfn::ErrorFn,
    initialization,
    layer::{
        Module, ModuleData, activation::Activation, chain::Chain, conv2d::Conv2D, dropout::Dropout,
        flatten::Flatten, linear::Linear,
    },
    optimizer::{
        Optimizer,
        sgd::{SGD, SGDWithMomentum},
    },
    tensor::tensor::{Tensor, TensorFloat},
};
//use tikv_jemallocator::Jemalloc;
//#[global_allocator]
//static GLOBAL: Jemalloc = Jemalloc;

//use mimalloc::MiMalloc;

//#[global_allocator]
//static GLOBAL: MiMalloc = MiMalloc;

#[derive(Serialize, Deserialize)]
struct ModelParameters<T: TensorFloat> {
    params: Vec<Tensor<T>>,
}

fn save_model<T: TensorFloat + Serialize>(model: &Chain<T>, path: &str) -> std::io::Result<()> {
    let params = ModelParameters {
        params: model.parameters(),
    };
    let serialized = serde_json::to_string(&params).unwrap();
    let mut file = File::create(path)?;
    file.write_all(serialized.as_bytes())?;
    Ok(())
}

fn load_model<T: TensorFloat + for<'de> Deserialize<'de>>(
    model: &mut Chain<T>,
    path: &str,
) -> std::io::Result<()> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let loaded_params: ModelParameters<T> = serde_json::from_str(&contents).unwrap();
    let model_params = model.parameters_mut();
    for (p, loaded_p) in model_params.into_iter().zip(loaded_params.params) {
        *p = loaded_p;
    }
    Ok(())
}

#[derive(Clone)]
struct EvalMetrics<T: TensorFloat> {
    accuracy: T,
    loss: T,

    // Per-class metrics
    per_class_accuracy: Vec<T>,
    per_class_precision: Vec<T>,
    per_class_recall: Vec<T>,
    per_class_f1: Vec<T>,

    // Macro-averaged (średnia z wszystkich klas)
    macro_precision: T,
    macro_recall: T,
    macro_f1: T,

    // Weighted-averaged (ważona liczbą sampli)
    weighted_precision: T,
    weighted_recall: T,
    weighted_f1: T,

    // Optional
    confusion_matrix: Option<Vec<Vec<usize>>>,
}

fn evaluate<T: TensorFloat + std::iter::Sum>(
    model: &mut Chain<T>,
    test_logits: &Vec<Tensor<T>>,
    test_labels: &Vec<Tensor<T>>,
    compute_confusion: bool,
) -> EvalMetrics<T> {
    model.inference();
    let num_classes = 10;

    let mut correct = 0;
    let mut total_loss = T::zero();

    // Per-class counters
    let mut true_positives = vec![0_usize; num_classes]; // poprawnie przewidziane jako klasa i
    let mut false_positives = vec![0_usize; num_classes]; // błędnie przewidziane jako klasa i
    let mut false_negatives = vec![0_usize; num_classes]; // pominięte (true=i, pred≠i)
    let mut total_per_class = vec![0_usize; num_classes]; // wszystkie prawdziwe klasy i

    let mut n = 0;
    // Confusion matrix [true][pred]
    let mut confusion: Vec<Vec<usize>> = vec![vec![0; num_classes]; num_classes];

    for i in (0..test_logits.len()).progress() {
        let y_pred = test_logits[i].clone();
        let y_true = &test_labels[i];

        let loss = tenebris::errorfn::softmax_with_crossentropy::SoftmaxCrossEntropy::new()
            .compute(&y_pred, y_true)
            .unwrap()
            .0;
        //let loss = cross_entropy_loss(&y_pred, y_true).0;
        total_loss = total_loss + loss;

        for j in 0..y_true.shape()[1] {
            let pred_idx = y_pred
                .select(1, j)
                .unwrap()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap();

            let true_idx = y_true
                .select(1, j)
                .unwrap()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap();

            // Update counters
            total_per_class[true_idx] += 1;

            if pred_idx == true_idx {
                correct += 1;
                true_positives[true_idx] += 1;
            } else {
                false_positives[pred_idx] += 1; // błędnie przewidziano jako pred_idx
                false_negatives[true_idx] += 1; // pominięto prawdziwą klasę true_idx
            }

            if compute_confusion {
                confusion[true_idx][pred_idx] += 1;
            }
            n += 1;
        }
    }

    let n = T::from(n).unwrap();
    let accuracy = T::from(correct).unwrap() / n;
    let avg_loss = total_loss / n;

    let mut per_class_accuracy = Vec::new();
    let mut per_class_precision = Vec::new();
    let mut per_class_recall = Vec::new();
    let mut per_class_f1 = Vec::new();

    for i in 0..num_classes {
        //accuracy dla klasy i
        let acc = if total_per_class[i] == 0 {
            T::zero()
        } else {
            T::from(true_positives[i]).unwrap() / T::from(total_per_class[i]).unwrap()
        };
        per_class_accuracy.push(acc);

        // precision: TP / (TP + FP)
        let tp = T::from(true_positives[i]).unwrap();
        let fp = T::from(false_positives[i]).unwrap();
        let precision = if true_positives[i] + false_positives[i] == 0 {
            T::zero()
        } else {
            tp / (tp + fp)
        };
        per_class_precision.push(precision);

        // recall: TP / (TP + FN)
        let fn_val = T::from(false_negatives[i]).unwrap();
        let recall = if true_positives[i] + false_negatives[i] == 0 {
            T::zero()
        } else {
            tp / (tp + fn_val)
        };
        per_class_recall.push(recall);

        // F1: 2 * (precision * recall) / (precision + recall)
        let f1 = if precision + recall == T::zero() {
            T::zero()
        } else {
            (precision * recall * T::from(2).unwrap()) / (precision + recall)
        };
        per_class_f1.push(f1);
    }

    let macro_precision =
        per_class_precision.iter().copied().sum::<T>() / T::from(num_classes).unwrap();
    let macro_recall = per_class_recall.iter().copied().sum::<T>() / T::from(num_classes).unwrap();
    let macro_f1 = per_class_f1.iter().copied().sum::<T>() / T::from(num_classes).unwrap();

    let mut weighted_precision = T::zero();
    let mut weighted_recall = T::zero();
    let mut weighted_f1 = T::zero();

    for i in 0..num_classes {
        let weight = T::from(total_per_class[i]).unwrap() / n;
        weighted_precision = weighted_precision + per_class_precision[i] * weight;
        weighted_recall = weighted_recall + per_class_recall[i] * weight;
        weighted_f1 = weighted_f1 + per_class_f1[i] * weight;
    }
    model.training();

    EvalMetrics {
        accuracy,
        loss: avg_loss,
        per_class_accuracy,
        per_class_precision,
        per_class_recall,
        per_class_f1,
        macro_precision,
        macro_recall,
        macro_f1,
        weighted_precision,
        weighted_recall,
        weighted_f1,
        confusion_matrix: if compute_confusion {
            Some(confusion)
        } else {
            None
        },
    }
}

fn print_eval<T: TensorFloat>(metrics: EvalMetrics<T>, epoch: usize, epochs: usize, train_acc: T) {
    println!("Epoch {}/{}", epoch, epochs);
    println!("  Train Accuracy: {:.2}%", train_acc);
    println!(
        "  Test Accuracy:  {:.2}%",
        metrics.accuracy * T::from(100.0).unwrap()
    );
    println!("  AVG Test Loss:  {:.4}", metrics.loss);
    println!("  Macro F1:       {:.4}", metrics.macro_f1);

    if epoch % 10 == 0 {
        println!("\n  Macro-averaged metrics:");
        println!("    Precision: {:.4}", metrics.macro_precision);
        println!("    Recall:    {:.4}", metrics.macro_recall);
        println!("    F1-Score:  {:.4}", metrics.macro_f1);

        println!("\n  Weighted-averaged metrics:");
        println!("    Precision: {:.4}", metrics.weighted_precision);
        println!("    Recall:    {:.4}", metrics.weighted_recall);
        println!("    F1-Score:  {:.4}", metrics.weighted_f1);
    }
}

fn print_long<T: TensorFloat>(
    final_metrics: EvalMetrics<T>,
    epoch: usize,
    epochs: usize,
    train_acc: T,
) {
    print_eval(final_metrics.clone(), epoch, epochs, train_acc);
    if let Some(cm) = final_metrics.confusion_matrix {
        println!("\nConfusion Matrix:");
        println!("True\\Pred   0    1    2    3    4    5    6    7    8    9");

        for (i, row) in cm.iter().enumerate() {
            print!("{:8}", i);
            for val in row {
                print!(" {:4}", val);
            }
            println!();
        }
    }
    println!("\nPer-class metrics:");
    println!(
        "{:<12} {:>8} {:>10} {:>8} {:>8}",
        "Class", "Accuracy", "Precision", "Recall", "F1-Score"
    );
    println!("{:-<52}", "");

    let class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ];

    for i in 0..10 {
        println!(
            "{:<12} {:>7.2}% {:>9.4} {:>7.4} {:>7.4}",
            class_names[i],
            final_metrics.per_class_accuracy[i] * T::from(100.0).unwrap(),
            final_metrics.per_class_precision[i],
            final_metrics.per_class_recall[i],
            final_metrics.per_class_f1[i]
        );
    }

    println!("{:-<52}", "");
    println!(
        "{:<12} {:>7.2}% {:>9.4} {:>7.4} {:>7.4}",
        "Macro avg",
        final_metrics.accuracy * T::from(100.0).unwrap(),
        final_metrics.macro_precision,
        final_metrics.macro_recall,
        final_metrics.macro_f1
    );
    println!(
        "{:<12} {:>7.2}% {:>9.4} {:>7.4} {:>7.4}",
        "Weighted avg",
        final_metrics.accuracy * T::from(100.0).unwrap(),
        final_metrics.weighted_precision,
        final_metrics.weighted_recall,
        final_metrics.weighted_f1
    );
}

fn load_cifar_batch<T: TensorFloat>(
    path: &Path,
    dataset_count: usize,
) -> (Vec<Tensor<T>>, Vec<Tensor<T>>) {
    let mut file = File::open(path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let mut images = vec![];
    let mut labels = vec![];

    for i in 0..dataset_count {
        let start = i * 3073;
        let mut label_tensor = Tensor::<T>::zeros([10, 1]).unwrap();
        label_tensor[&[buffer[start] as usize, 0]] = T::one();
        labels.push(label_tensor);
        let image_raw_data = buffer[start + 1..start + 3073]
            .iter()
            .map(|x| {
                ((T::from(*x).unwrap() * T::from(2.0).unwrap()) / T::from(255).unwrap()) - T::one()
            })
            .collect();
        let image_tensor = unsafe { Tensor::new_raw([3, 1024], image_raw_data, [1024, 1]) };
        let image_tensor = image_tensor
            .make_contiguous()
            .unwrap()
            .reshape(&[3, 32, 32, 1])
            .unwrap();
        images.push(image_tensor);
    }

    (images, labels)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    type T = f32;
    println!("Loading CIFAR-10 dataset...");
    let dir_path = env::current_dir()?;
    println!("Project directory: {:?}", dir_path);
    let data_path = dir_path.as_path().join("cifar-10-batches-bin");
    println!("Data path: {:?}", data_path);
    if !data_path.exists() {
        println!("Downloading dataset...");
        let client = Client::builder()
            .timeout(Duration::from_secs(300)) // Ustaw timeout na 5 minut
            .build()?;
        let tarball = client
            .get("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")
            .send()?
            .bytes()?;
        let mut archive = Archive::new(GzDecoder::new(&tarball[..]));
        archive.unpack(dir_path.clone())?;
    }

    let mut images_train: Vec<Tensor<T>> = Vec::new();
    let mut labels_train = Vec::new();
    for i in 1..=5 {
        let (images, labels) =
            load_cifar_batch(&data_path.join(format!("data_batch_{}.bin", i)), 10000);
        images_train.extend(images);
        labels_train.extend(labels);
    }

    let (images_test, labels_test): (Vec<Tensor<T>>, Vec<Tensor<T>>) =
        load_cifar_batch(&data_path.join("test_batch.bin"), 10000);
    println!("Dataset loaded.");

    //DEBUG - save image; zdjęcie będzie w dziwnej kolorystyce -- shift do ujemnych przy ładowaniu
    //=> clamp
    /*
    let tensor = &images_train[0]; // [3, 32, 32, 1]
    let data = tensor.get_data();

    let width = 32;
    let height = 32;
    let channels = 3;

    let mut img = RgbImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let base = channels * (x + width * y);
            let r = (data[base + 0] as f64 * 255.0).clamp(0.0, 255.0) as u8;
            let g = (data[base + 1] as f64 * 255.0).clamp(0.0, 255.0) as u8;
            let b = (data[base + 2] as f64 * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    img.save("debug_image.png").unwrap();

    let mut img = RgbImage::new(32, 32);
    for y in 0..32 {
        for x in 0..32 {
            let idx = y * 32 + x;
            let r = data[idx] as u8;
            let g = data[1024 + idx] as u8;
            let b = data[2048 + idx] as u8;
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    img.save("debug_image2.png").unwrap();
    // */
    let init = initialization::he::He::new();

    let mut model = Chain::new(vec![
        // 32x32x3 -> 32x32x32
        Box::new(Conv2D::<T>::new(3, 32, (3, 3), (1, 1), (1, 1), &init)?),
        Box::new(Activation::elu(1.0)), // ELU > ReLU dla małych sieci
        Box::new(Dropout::new(0.1)),
        // 32x32x32 -> 16x16x64
        Box::new(Conv2D::<T>::new(32, 64, (3, 3), (2, 2), (1, 1), &init)?),
        Box::new(Activation::elu(1.0)),
        Box::new(Dropout::new(0.2)),
        // 16x16x64 -> 8x8x64
        Box::new(Conv2D::<T>::new(64, 128, (3, 3), (2, 2), (1, 1), &init)?),
        Box::new(Activation::elu(1.0)),
        Box::new(Dropout::new(0.3)),
        // Flatten -> 8*8*64 = 4096
        Box::new(Flatten::new()),
        Box::new(Linear::<T>::new_with_init(8 * 8 * 128, 512, &init)?),
        Box::new(Activation::elu(1.0)),
        Box::new(Dropout::new(0.5)),
        // 512 -> 128
        Box::new(Linear::<T>::new_with_init(512, 128, &init)?),
        Box::new(Activation::elu(1.0)),
        Box::new(Dropout::new(0.5)),
        // Output
        Box::new(Linear::<T>::new_with_init(128, 10, &init)?),
    ]);

    let mut optimizer = SGDWithMomentum::new(0.01, 0.9, 0.01); // wyższy LR
    //
    let epochs = 50; // Więcej epok, bo mniej sampli per epoch
    let samples_per_epoch = 50000;
    //let minibatch_size = 32;
    let _batch_size = 64;

    //println!("\nLoading model...");
    //load_model(&mut model, "model_75.228.json").unwrap();
    //println!("Model loaded.");

    println!("===============================================================");
    println!("Evaluating initial model...");

    let (images, labels) = tenebris::batch_builder::BatchBuilder::build_batches_with_labels(
        &images_test,
        &labels_test,
        false,
        128,
    )
    .unwrap();

    let mut logits = Vec::new();

    model.inference();

    let timer = Instant::now();
    for b in images.iter().progress() {
        let x: Tensor<f32> = model.forward(b.clone(), false).unwrap();
        logits.push(x);
    }
    println!(
        "Forward time for test set: {:.2}ms",
        timer.elapsed().as_secs_f32() * 1000.0
    );

    model.training();

    let metrics = evaluate(&mut model, &logits, &labels, false);
    println!(
        "Forward + evaluate time for test set: {:.2}ms",
        timer.elapsed().as_secs_f32() * 1000.0
    );
    print_eval(metrics, 0, epochs, 0.0);
    println!("Parameter value stats in Modules:");
    for (num, param) in model.parameters().iter().enumerate() {
        let data = param;
        let mut sum = 0.0;
        let mut max_abs = 0.0;
        for x in data.iter() {
            let v = x;
            sum += v;
            let abs_v = v.abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
        }
        let mean = sum / data.get_data().len() as T;
        let l2_norm: T = data
            .iter()
            .map(|x| {
                let v = x.to_f32().unwrap();
                v * v
            })
            .sum::<T>()
            .sqrt();
        println!("Module[{num}]: mean={mean:.6}, max|W|={max_abs:.6}, l2={l2_norm:.6}");
    }

    println!("===============================================================");
    model.clear_grad();
    // */
    // /*
    let mut best_accuracy_train = 0.0;

    let softmax_with_crossentropy =
        tenebris::errorfn::softmax_with_crossentropy::SoftmaxCrossEntropy::new();

    let augmentation = AugmentationOptions::new(true, 3, 32, 32);
    for epoch in 1..(epochs + 1) {
        let mut correct = 0;
        // Losuj indeksy
        let mut indices: Vec<usize> = (0..images_train.len()).collect();
        let mut rng = rng();
        indices.shuffle(&mut rng);

        let mut total_loss = 0.0;

        let mut time_forward = Duration::from_secs(0);
        let mut time_loss = Duration::from_secs(0);
        let mut time_backward = Duration::from_secs(0);
        let mut time_optim = Duration::from_secs(0);

        let img_label_touple_batch =
            tenebris::batch_builder::BatchBuilder::build_batches_with_labels2(
                &images_train,
                &labels_train,
                true,
                _batch_size,
                augmentation.clone(),
            )
            .unwrap();

        let start = Instant::now();
        let mut i = 0;
        if epoch == 25 || epoch == 35 {
            optimizer.set_learning_rate(optimizer.get_learning_rate() * 0.1);
            println!(
                "--- learning rate decay! now: {:.6} ---",
                optimizer.get_learning_rate()
            );
        }
        for b in img_label_touple_batch.iter().progress() {
            let (img, lab) = b;

            //println!("IMG SIZE: {:?}", img.shape());

            let start = Instant::now();
            let y: Tensor<f32> = model.forward(img.clone(), true).unwrap();
            time_forward += start.elapsed();
            //println!("y: {y}");
            //println!("t: {lab}");

            let start = Instant::now();
            let (loss, loss_grad) = softmax_with_crossentropy.compute(&y, &lab)?;
            total_loss += loss;
            time_loss += start.elapsed();
            //println!("g: {loss_grad}");

            for j in 0..y.shape()[1] {
                let pred_idx = y
                    .select(1, j)
                    .unwrap()
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap();

                let true_idx = lab
                    .select(1, j)
                    .unwrap()
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap();

                //println!("y: {y}");
                //println!("t: {lab}");
                // Update counters
                //println!("pred_idx: {pred_idx}");
                //println!("true_idx: {true_idx}");

                if pred_idx == true_idx {
                    correct += 1;
                } else {
                }
            }

            let start = Instant::now();
            model.backward(ModuleData::Single(loss_grad)).unwrap();
            time_backward += start.elapsed();

            //if i % 32 == 0 || i % (samples_per_epoch - 1) == 0 {
            let start = Instant::now();
            optimizer.step(&mut model).unwrap();
            time_optim += start.elapsed();
            //}
            i = i + 1;
        }
        println!(
            "time for backprop fast: {:4}",
            start.elapsed().as_secs_f32()
        );
        println!("==========================Epoch {:4}", epoch,);

        println!(
            "|-> Train Avg Loss: {:.6}",
            total_loss / samples_per_epoch as T
        );

        let train_accuracy = 100.0 * (correct as T / samples_per_epoch as T);
        println!("|--> Train Accuracy: {:}%", train_accuracy);
        if train_accuracy > best_accuracy_train {
            best_accuracy_train = train_accuracy;
            save_model(&model, &format!("model_{best_accuracy_train}.json")).unwrap();
        }

        // eval co x epok
        if epoch % 1 == 0 {
            println!("===============================================================");
            println!("Evaluating model...");
            let (images, labels) =
                tenebris::batch_builder::BatchBuilder::build_batches_with_labels(
                    &images_test,
                    &labels_test,
                    false,
                    128,
                )
                .unwrap();
            model.inference();
            let mut y = Vec::new();
            let el = Instant::now();
            for b in images.iter().progress() {
                let x: Tensor<f32> = model.forward(b.clone(), false).unwrap();
                y.push(x);
            }
            let metrics = evaluate(&mut model, &y, &labels, false);
            println!("FORWARD: {:?}", el.elapsed());
            model.training();
            //print_eval(metrics, epoch, epochs, train_accuracy);
            print_long(metrics, epoch, epochs, train_accuracy);
            println!("==================================================================");
        }
        // TODO: FIX IT chyba fixed
        println!("---DEBUG---");
        println!("Parameter value stats in Modules:");
        for (num, param) in model.parameters().iter().enumerate() {
            let data = param;
            let mut sum = 0.0;
            let mut max_abs = 0.0;
            for x in data.iter() {
                let v = x;
                sum += v;
                let abs_v = v.abs();
                if abs_v > max_abs {
                    max_abs = abs_v;
                }
            }
            let mean = sum / data.get_data().len() as T;
            let l2_norm: T = data
                .iter()
                .map(|x| {
                    let v = x.to_f32().unwrap();
                    v * v
                })
                .sum::<T>()
                .sqrt();
            println!("Module[{num}]: mean={mean:.6}, max|W|={max_abs:.6}, l2={l2_norm:.6}");
        }
        println!(
            "Forward: {:.2}ms, Loss: {:.2}ms, Backward: {:.2}ms, Optim: {:.2}ms",
            time_forward.as_secs_f32() * 1000.0,
            time_loss.as_secs_f32() * 1000.0,
            time_backward.as_secs_f32() * 1000.0,
            time_optim.as_secs_f32() * 1000.0,
        );
        println!("---DEBUG-END---");
        model.clear_grad();
    }

    println!("Training finished.");

    println!("\nSaving model...");
    save_model(&model, "model.json").unwrap();
    println!("Model saved.");

    println!("\nLoading model...");
    load_model(&mut model, "model.json").unwrap();
    println!("Model loaded.");

    println!("\nEvaluating final model...");
    // */
    Ok(())
}
