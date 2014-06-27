package ml.shifu.core.request.processor;

import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.core.request.RequestObject;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;

public class InitRequestProcessorTest {

    @Test
    public void test() throws IOException {


        ObjectMapper jsonMapper = new ObjectMapper();
        RequestObject req = jsonMapper.readValue(new File("src/test/resources/models/wdbc/CSVDataDictionaryInitializer/request.json"), RequestObject.class);

        CreateDataDictionaryRequestProcessor processor = new CreateDataDictionaryRequestProcessor();
        processor.run(req);
    }
}
