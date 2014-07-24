package ml.shifu.core.request.processor;

import ml.shifu.core.request.RequestObject;
import ml.shifu.core.request.processor.deprecated.CreateMiningSchemaRequestProcessor;
import ml.shifu.core.util.JSONUtils;
import org.testng.annotations.Test;

import java.io.File;

public class CreateMiningSchemaRequestProcessorTest {

    @Test
    public void test() throws Exception {
        RequestObject req = JSONUtils.readValue(new File("src/test/resources/models/wdbc/CreateMiningSchema/request.json"), RequestObject.class);

        CreateMiningSchemaRequestProcessor processor = new CreateMiningSchemaRequestProcessor();
        processor.run(req);
    }
}
