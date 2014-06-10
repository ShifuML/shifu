package ml.shifu.shifu.request.processor;

import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.shifu.request.RequestObject;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;

public class StatsRequestProcessorTest {

    @Test
    public void test() throws Exception{


        ObjectMapper jsonMapper = new ObjectMapper();
        RequestObject req = jsonMapper.readValue(new File("src/test/resources/request/LocalSingleStatsRequest.json"), RequestObject.class);

        StatsRequestProcessor processor = new StatsRequestProcessor();
        processor.run(req);


    }
}
