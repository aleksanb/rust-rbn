extern crate ndarray;
extern crate rand;
extern crate rusty_machine;
use rusty_machine::learning::logistic_reg::LogisticRegressor;

#[macro_use]
mod utils;
mod tasks;
mod rbn;
mod rbn_system;

use tasks::{Task, TaskType};

fn calculate_accuracies(n_samples: usize) -> Vec<f64> {
    let training_size = 4000;
    let test_size = 200;
    let n_nodes = 70;
    let connectivity = 3;

    let training_set = Task::new(TaskType::TemporalParity,
                                 training_size,
                                 3);
    let testing_set = Task::new(TaskType::TemporalParity,
                                test_size,
                                3);


    (0..n_samples)
        .map(|_| {
            let mut rbn_system = rbn_system::ReservoirSystem {
                readout_layer: LogisticRegressor::default(),
                rbn_reservoir: rbn::RBN::new(n_nodes, connectivity, n_nodes / 2),
            };
            rbn_system.train_on(&training_set);
            rbn_system.test_on(&testing_set)
        })
        .collect()
}

fn main() {
    let accuracies = calculate_accuracies(50);
    p!(accuracies);
}
