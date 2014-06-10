package ml.shifu.shifu.request.processor;

import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.JSONUtils;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;

public class CreateModelElementRequestProcessorTest {

    @Test
    public void test() throws Exception{
        RequestObject req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/CreateModelElement/request.json"), RequestObject.class);

        CreateModelElementRequestProcessor processor = new CreateModelElementRequestProcessor();
        processor.run(req);
    }
}
