package ml.shifu.shifu.di.service;

import com.google.inject.Inject;
import ml.shifu.shifu.container.PMMLDataSet;
import ml.shifu.shifu.di.spi.Trainer;
import ml.shifu.shifu.util.Params;

public class TrainingService {

    private Trainer trainer;

    @Inject
    public TrainingService(Trainer trainer) {
        this.trainer = trainer;
    }


    public void exec(PMMLDataSet dataSet, Params rawParams) throws Exception {
        trainer.train(dataSet, rawParams);
    }
}
