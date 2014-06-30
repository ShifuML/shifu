package ml.shifu.plugin.encog.trainer;

import ml.shifu.core.container.ShifuRequest;
import ml.shifu.core.request.RequestDispatcher;
import ml.shifu.core.util.JSONUtils;
import org.testng.annotations.Test;

import java.io.File;

public class EncogNNTrainerTest {

    @Test
    public void test8() throws Exception {
        ShifuRequest req = JSONUtils.readValue(new File("src/test/resources/trainer/ExecTrainRequest.json"), ShifuRequest.class);
        RequestDispatcher.dispatch(req);

    }
}
