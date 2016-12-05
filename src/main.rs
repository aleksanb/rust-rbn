extern crate ndarray;
extern crate rand;
extern crate rusty_machine;
use rusty_machine::prelude::*;
use rusty_machine::learning::logistic_reg::LogisticRegressor;

#[macro_use]
mod utils;
mod tasks;
mod rbn;

use tasks::{Task, TaskType};

fn main() {
    let n_samples = 4000;
    let n_nodes = 400;
    let rbn = rbn::RBN::new(n_nodes, 3, n_nodes / 2);

    let training_set = Task::new(TaskType::TemporalParity,
                                        n_samples,
                                        3);
    let matrix = Matrix::new(
        n_samples, n_nodes,
        l![i as f64, i <- rbn.execute(&training_set.input)]);
    let training_output = Vector::new(l![i as f64, i <- training_set.output]);

    let mut lin_mod = LogisticRegressor::default();
    lin_mod.train(&matrix, &training_output).expect("Training failed");


    let testing_samples = 200;
    let testing_set = Task::new(TaskType::TemporalParity,
                                testing_samples,
                                3);
    let test_result = Matrix::new(
        testing_samples, n_nodes,
        l![i as f64, i <- rbn.execute(&testing_set.input)]);
    let predictions = lin_mod.predict(&test_result).expect("Couldn't predict dataset");


    let mut n_errors = 0;
    for (est, corr) in predictions.iter().zip(testing_set.output) {
        if (est > &0.5) as u8 != corr {
            n_errors += 1;
        }
    }

    let accuracy = 1.0 - ((n_errors as f64) / (predictions.size() as f64));
    p!(n_errors);
    p!(accuracy);
}
