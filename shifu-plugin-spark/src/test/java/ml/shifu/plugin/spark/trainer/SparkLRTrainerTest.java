package ml.shifu.plugin.spark.trainer;

import java.io.File;

import ml.shifu.core.container.ShifuRequest;
import ml.shifu.core.request.RequestDispatcher;
import ml.shifu.core.util.JSONUtils;

import org.testng.annotations.Test;

public class SparkLRTrainerTest {

    @Test
    public void testSparkLRTrainer() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/trainer/sparkLR/SparkLRExecTrainRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);
    }
}
