package ml.shifu.plugin.pig;

import ml.shifu.core.request.Request;
import ml.shifu.core.request.RequestDispatcher;
import ml.shifu.core.util.JSONUtils;
import org.testng.annotations.Test;

import java.io.File;

public class PigStatsRequestProcessorTest {

    //@Test
    public void test1() throws Exception {
        RequestDispatcher.dispatch(JSONUtils.readValue(new File(
                "src/test/resources/request/pigstats.json"), Request.class));
    }

}
