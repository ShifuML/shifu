package ml.shifu.plugin.encog.trainer;

import ml.shifu.core.request.Request;
import ml.shifu.core.request.RequestDispatcher;
import ml.shifu.core.util.JSONUtils;

import org.apache.commons.io.FileUtils;
import org.testng.annotations.AfterTest;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;

public class EncogNNTrainerTest {

    @Test
    public void test5() throws Exception {
        Request req = JSONUtils.readValue(new File("src/test/resources/trainer/train.json"), Request.class);
        RequestDispatcher.dispatch(req);
    }
    
    @AfterTest
    public void tearDown() throws IOException {
        FileUtils.deleteQuietly(new File("src/test/resources/trainer/model_output.xml"));
        FileUtils.deleteDirectory(new File("src/test/resources/models"));
    }
}
