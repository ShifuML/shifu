package ml.shifu.plugin.mahout.trainer;

import ml.shifu.core.container.ShifuRequest;
import ml.shifu.core.request.RequestDispatcher;
import ml.shifu.core.util.JSONUtils;
import org.testng.annotations.Test;

import java.io.File;

public class MahoutNNTrainerTest {

    @Test
    public void testMahoutNNTrainer() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/trainer/MahoutNNExecTrainRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);
    }
}
