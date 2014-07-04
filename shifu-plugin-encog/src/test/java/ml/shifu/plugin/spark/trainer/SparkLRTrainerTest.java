package ml.shifu.plugin.spark.trainer;

import ml.shifu.core.container.ShifuRequest;
import ml.shifu.core.request.RequestDispatcher;
import ml.shifu.core.util.JSONUtils;
import org.testng.annotations.Test;

import java.io.File;

public class SparkLRTrainerTest {

    @Test
    public void testSparkLRTrainer() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/trainer/SparkLRExecTrainRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);
    }
}
