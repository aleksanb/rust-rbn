#[macro_use]
extern crate clap;
use clap::App;

extern crate ndarray;
extern crate rand;
extern crate rusty_machine;
extern crate rayon;
use rusty_machine::learning::logistic_reg::LogisticRegressor;
use rayon::prelude::*;

#[macro_use]
mod utils;
mod tasks;
mod rbn;
mod rbn_system;

use tasks::Task;

struct Experiment {
    training_size: usize,
    test_size: usize,
    n_nodes: usize,
    connectivity: usize,
    n_samples: usize,
    task_window_size: usize,
}

fn calculate_accuracies(ex: Experiment) -> Vec<f64> {
    let training_set = Task::new(ex.training_size,
                                 ex.task_window_size.clone());
    let testing_set = Task::new(ex.test_size,
                                ex.task_window_size.clone());


    let mut output_vec = vec![0f64; ex.n_samples];
    let nums = (0..ex.n_samples).collect::<Vec<usize>>();
    nums.par_iter()
        .map(|_| {
            let mut rbn_system = rbn_system::ReservoirSystem {
                readout_layer: LogisticRegressor::default(),
                rbn_reservoir: rbn::RBN::new(ex.n_nodes, ex.connectivity, ex.n_nodes / 2),
            };
            rbn_system.train_on(&training_set);
            rbn_system.test_on(&testing_set)
        })
        .collect_into(&mut output_vec);

    output_vec
}

fn main() {
    let yaml = load_yaml!("cli.yml");
    let matches = App::from_yaml(yaml).get_matches();

    let experiment = Experiment {
        training_size: 4000,
        test_size: 200,
        n_nodes: 70,
        connectivity: 3,
        task_window_size: 3,
        n_samples: 100,
    };

    let accuracies = calculate_accuracies(experiment);
    p!(accuracies);
}
