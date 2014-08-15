package ml.shifu.plugin.spark.norm;
import java.io.File;

import org.testng.Assert;
import org.testng.annotations.Test;

import ml.shifu.core.request.Request;
import ml.shifu.core.util.JSONUtils;
import ml.shifu.plugin.spark.norm.SparkModelTransformRequestProcessor;


public class SparkModelTransformRequestProcessorTest {

    @Test
    public void test() throws Exception {
        SparkModelTransformRequestProcessor strp= new SparkModelTransformRequestProcessor();
        Request req=  JSONUtils.readValue(new File("src/test/resources/5_transformexec.json"), Request.class); 
        strp.exec(req);
    }
}
