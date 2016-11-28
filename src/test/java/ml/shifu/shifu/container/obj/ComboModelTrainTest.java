package ml.shifu.shifu.container.obj;

import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.JSONUtils;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhanhu on 11/18/16.
 */
public class ComboModelTrainTest {

    @Test
    public void testSerDeser() throws IOException {
        ComboModelTrain inst = new ComboModelTrain();

        inst.setUidColumnName("txnId");
        List<ModelTrainConf> modelTrainConfList = new ArrayList<ModelTrainConf>();
        modelTrainConfList.add(createModelTrainConf(ModelTrainConf.ALGORITHM.NN));
        modelTrainConfList.add(createModelTrainConf(ModelTrainConf.ALGORITHM.GBT));

        inst.setModelTrainConfList(modelTrainConfList);
        inst.setFusionModelTrainConf(createModelTrainConf(ModelTrainConf.ALGORITHM.LR));

        JSONUtils.writeValue(new File("src/test/resources/example/combotrain/ComboTrain.json"), inst);

        ComboModelTrain anotherInst =
                JSONUtils.readValue(new File("src/test/resources/example/combotrain/ComboTrain.json"),
                        ComboModelTrain.class);
        Assert.assertEquals(inst, anotherInst);
    }

    private ModelTrainConf createModelTrainConf(ModelTrainConf.ALGORITHM alg) {
        ModelTrainConf trainConf = new ModelTrainConf();

        trainConf.setAlgorithm(alg.name());
        trainConf.setEpochsPerIteration(1);
        trainConf.setParams(ModelTrainConf.createParamsByAlg(alg, trainConf));
        trainConf.setNumTrainEpochs(100);
        if(ModelTrainConf.ALGORITHM.NN.equals(alg)) {
            trainConf.setNumTrainEpochs(200);
        } else if(ModelTrainConf.ALGORITHM.SVM.equals(alg)) {
            trainConf.setNumTrainEpochs(100);
        } else if(ModelTrainConf.ALGORITHM.RF.equals(alg)) {
            trainConf.setNumTrainEpochs(20000);
        } else if(ModelTrainConf.ALGORITHM.GBT.equals(alg)) {
            trainConf.setNumTrainEpochs(20000);
        } else if(ModelTrainConf.ALGORITHM.LR.equals(alg)) {
            trainConf.setNumTrainEpochs(100);
        }
        trainConf.setBaggingWithReplacement(true);

        return trainConf;
    }
}
