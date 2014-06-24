package ml.shifu.shifu.request;

import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.shifu.util.Params;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class RequestObjectTest {

    @Test
    public void test() throws IOException {
        RequestObject req = new RequestObject();

        req.setRequestType("test");
        req.setExecutionMode(RequestObject.ExecutionMode.LOCAL_SINGLE);


        Params params = new Params();

        params.put("numBins", 10);
        params.put("posTags", Arrays.asList("Iris-setosa", "Iris-versicolor"));
        params.put("negTags", Arrays.asList("Iris-virginica"));


        req.setGlobalParams(params);

        Map<String, Params> fieldParamsMap = new HashMap<String, Params>();

        for (int i = 0; i < 5; i++) {
            Params p = new Params();
            p.put("key", "value");
            if (i == 0) {
                fieldParamsMap.put("$$default", p);
            } else {
                fieldParamsMap.put("field" + i, p);
            }
        }

        params.setFieldParamsMap(fieldParamsMap);


        ObjectMapper jsonMapper = new ObjectMapper();

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(new File("request.json"), req);


        RequestObject reqObj = jsonMapper.readValue(new File("request.json"), RequestObject.class);

        //Assert.assertEquals(((List<String>)reqObj.getGlobalParameters().get("posTags")).size(), 2);
    }

}
