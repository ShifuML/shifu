package ml.shifu.shifu.request.processor;

import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.JSONUtils;
import org.testng.annotations.Test;

import java.io.File;

public class TransformationRequestProcessorTest {
    @Test
    public void test() throws Exception {

        RequestObject req = JSONUtils.readValue(new File("src/test/resources/request/TransformationRequest.json"), RequestObject.class);

        TransformationRequestProcessor processor = new TransformationRequestProcessor();
        processor.run(req);
    }
}
