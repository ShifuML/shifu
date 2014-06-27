package ml.shifu.core.di.service;

import com.google.inject.Inject;
import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.di.spi.Trainer;
import ml.shifu.core.util.Params;

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
