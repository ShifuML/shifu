package ml.shifu.core.request;

import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.core.util.JSONUtils;
import ml.shifu.core.util.Params;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class RequestTest {

   // @Test
    public void test() throws Exception {
        Request req = JSONUtils.readValue(new File("src/test/resources/request/newformat.json"), Request.class);

        Params p = req.getBindings().get(0).getParams().getFieldParams("id");

        Assert.assertEquals(req.getBindings().size(), 1);
    }

}
