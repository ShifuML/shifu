package ml.shifu.shifu.request.processor;

import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.JSONUtils;
import org.testng.annotations.Test;

import java.io.File;

public class CreateMiningSchemaRequestProcessorTest {

    @Test
    public void test() throws Exception{
        RequestObject req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/CreateMiningSchema/request.json"), RequestObject.class);

        CreateMiningSchemaRequestProcessor processor = new CreateMiningSchemaRequestProcessor();
        processor.run(req);
    }
}
