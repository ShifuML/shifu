package ml.shifu.plugin.spark.stats;

import java.io.File;

import ml.shifu.core.request.Request;
import ml.shifu.core.util.JSONUtils;
import org.testng.annotations.Test;

public class SparkCalcStatsRequestProcessorTest {

    @Test
    public void test() throws Exception {
        SparkCalcStatsRequestProcessor strp= new SparkCalcStatsRequestProcessor();
        Request req=  JSONUtils.readValue(new File("src/test/resources/stats/spark_stats.json"), Request.class); 
        strp.exec(req);
    }

}
