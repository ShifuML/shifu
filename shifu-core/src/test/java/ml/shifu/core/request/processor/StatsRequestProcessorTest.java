package ml.shifu.core.request.processor;

import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.core.request.RequestObject;
import org.testng.annotations.Test;

import java.io.File;

public class StatsRequestProcessorTest {

    @Test
    public void test() throws Exception {


        ObjectMapper jsonMapper = new ObjectMapper();
        RequestObject req = jsonMapper.readValue(new File("src/test/resources/models/wdbc/StatsLocal/request.json"), RequestObject.class);

        ExecStatsRequestProcessor processor = new ExecStatsRequestProcessor();
        processor.run(req);


    }
}
