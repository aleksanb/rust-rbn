use rusty_machine::learning::logistic_reg::LogisticRegressor;
use rusty_machine::learning::optim::grad_desc::GradientDesc;
use rusty_machine::prelude::*;
use rbn::RBN;
use tasks::Task;

#[derive(Debug)]
pub struct ReservoirSystem {
    pub readout_layer: LogisticRegressor<GradientDesc>,
    pub rbn_reservoir: RBN,
}

impl ReservoirSystem {
    pub fn train_on(&mut self, training_task: &Task) {
        let intermediate_states = Matrix::new(
            training_task.input.len(), self.rbn_reservoir.n_nodes,
            l![i as f64, i <- self.rbn_reservoir.execute(&training_task.input)]);

        let training_expected_output = Vector::new(
            l![*i as f64, i <- &training_task.output]);

        self.readout_layer.train(&intermediate_states, &training_expected_output)
            .expect("Training failed");
    }

    pub fn test_on(&mut self, testing_task: &Task) -> f64 {
        let intermediate_states = Matrix::new(
            testing_task.input.len(), self.rbn_reservoir.n_nodes,
            l![i as f64, i <- self.rbn_reservoir.execute(&testing_task.input)]);

        let predictions = self.readout_layer.predict(&intermediate_states)
            .expect("Couldn't predict dataset");


        let n_errors = predictions.iter()
            .zip(&testing_task.output)
            .map(|(est, corr)| if (est > &0.5) as u8 != *corr { 1 } else {0})
            .sum::<usize>();

        1.0 - ((n_errors as f64) / (predictions.size() as f64))
    }
}
