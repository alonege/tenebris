#[allow(unused_imports)]
use tenebris::{
    errorfn::{ErrorFn, mse::Mse},
    layer::{IntoModuleData, Module, activation::Activation, chain::Chain, linear::Linear},
    optimizer::{
        Optimizer,
        sgd::{SGD, SGDWithMomentum},
    },
    tensor::tensor::Tensor,
};

fn main() {
    let x_col1 = vec![0.0, 0.0, 1.0, 1.0];
    let x_col2 = vec![0.0, 1.0, 0.0, 1.0];

    let y_true_col = vec![0.0, 1.0, 1.0, 0.0];

    let num_samples = 4;

    let init = tenebris::initialization::glorot::Glorot;
    let l1 = Linear::<f32>::new_with_init(2, 3, &init).unwrap();
    let l2 = Linear::<f32>::new_with_init(3, 1, &init).unwrap();
    let mut model = Chain::new(vec![
        Box::new(l1),                 // Warstwa 1: 2 wejścia -> 3 neurony
        Box::new(Activation::tanh()), // Aktywacja
        Box::new(l2),                 // Warstwa 2: 3 neurony -> 1 wyjście
    ]);

    //let mut optimizer = SGDWithMomentum::new(0.1, 0.9);
    let mut optimizer = SGD::new(0.1);
    let epochs = 300;

    println!("Rozpoczynam trening sieci dla XOR (próbka po próbce)...");

    for epoch in 0..=epochs {
        let mut total_epoch_loss = 0.0_f32;

        for i in 0..num_samples {
            let x_sample = Tensor::new(vec![2, 1], vec![x_col1[i], x_col2[i]]).unwrap();

            let y_sample = Tensor::new(vec![1, 1], vec![y_true_col[i]]).unwrap();

            let y_pred = model.forward(x_sample, true).unwrap();
            let y_pred = &y_pred; // Oczekujemy jednego tensora wyjściowego

            let mse = Mse::new();
            let (loss, loss_grad) = mse.compute(y_pred, &y_sample).unwrap();
            total_epoch_loss += loss;

            let _ = model.backward(loss_grad.into_module_data());

            let _ = optimizer.step(&mut model);
            model.clear_grad();
        }
        //println!("PARAMS:");
        //for l in model.parameters() {
        //    println!("param: {:?}", l);
        //}

        // wypiszmy stratę
        if epoch % 10 == 0 {
            let avg_loss = total_epoch_loss / num_samples as f32;
            println!("Epoka: {:4}, Średnia strata: {:.6}", epoch, avg_loss);
        }
    }

    // weryfikacja po treningu
    println!("\nTrening zakończony. Predykcje końcowe:");

    let mut final_preds = vec![];
    for i in 0..num_samples {
        let x1 = x_col1[i];
        let x2 = x_col2[i];
        let y_t = y_true_col[i];

        // stwórzmy tensor inputu
        let x_sample = Tensor::new(vec![2, 1], vec![x1, x2]).unwrap();

        let pred_vec: Tensor<f32> = model.forward(x_sample, false).unwrap();
        let y_p = pred_vec.get(&[0, 0]).unwrap(); // Kształt wyjścia to [1, 1]
        final_preds.push(y_p);

        println!(
            "Wejście: [{}, {}], Oczekiwano: {}, Otrzymano: {:.4}",
            x1, x2, y_t, y_p
        );
    }

    let final_params = model.parameters();
    println!("\nParametry końcowe modelu");
    println!("Parametry: {:?}", final_params);
    for (i, param) in final_params.iter().enumerate() {
        println!("Parametr {}: Kształt: {:?}, Dane:", i, param.shape());
        println!("{:?}\n", param.get_data());
    }
    println!("--- Koniec parametrów ---");

    // sprawdźmy, czy model radzi sobie z danymi
    assert!(final_preds[0] < 0.2, "0 XOR 0 powinno być < 0.2");
    assert!(final_preds[1] > 0.8, "0 XOR 1 powinno być > 0.8");
    assert!(final_preds[2] > 0.8, "1 XOR 0 powinno być > 0.8");
    assert!(final_preds[3] < 0.2, "1 XOR 1 powinno być < 0.2");

    println!("\nAsercje pomyślne. Sieć nauczyła się XOR.");
}
