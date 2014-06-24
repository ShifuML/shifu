package ml.shifu.shifu.request.processor;

import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.JSONUtils;
import org.testng.annotations.Test;

import java.io.File;

public class CreateModelElementRequestProcessorTest {

    @Test
    public void test() throws Exception {
        RequestObject req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/CreateModelElement/request.json"), RequestObject.class);

        CreateModelElementRequestProcessor processor = new CreateModelElementRequestProcessor();
        processor.run(req);
    }
}
