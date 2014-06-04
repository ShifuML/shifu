package ml.shifu.shifu.request;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class RequestObjectTest  {

    @Test
    public void test() throws IOException {
        RequestObject req = new RequestObject();

        req.setAction("test");
        req.setExecutionMode(RequestObject.ExecutionMode.LOCAL_SINGLE);
        Map<String, String> bindings = new HashMap<String, String>();

        bindings.put("test_spi1", "test_impl1");
        bindings.put("test_spi2", "test_impl2");

        req.setBindings(bindings);

        Map<String, Object> parameters = new HashMap<String, Object>();

        parameters.put("numBins", 10);
        parameters.put("posTags", Arrays.asList("Iris-setosa", "Iris-versicolor"));
        parameters.put("negTags", Arrays.asList("Iris-virginica"));

        req.setParameters(parameters);

        ObjectMapper jsonMapper = new ObjectMapper();

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(new File("request.json"), req);


        RequestObject reqObj = jsonMapper.readValue(new File("request.json"), RequestObject.class);

        Assert.assertEquals(((List<String>)reqObj.getParameters().get("posTags")).size(), 2);
    }

}
