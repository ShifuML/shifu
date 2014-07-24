package ml.shifu.plugin.encog.trainer;

import ml.shifu.core.container.ShifuRequest;
import ml.shifu.core.request.Request;
import ml.shifu.core.request.RequestDispatcher;
import ml.shifu.core.util.JSONUtils;
import org.testng.annotations.Test;

import java.io.File;

public class EncogNNTrainerTest {

    @Test
    public void test5() throws Exception {
        Request req = JSONUtils.readValue(new File("src/test/resources/trainer/train.json"), Request.class);
        RequestDispatcher.dispatch(req);
    }
}
