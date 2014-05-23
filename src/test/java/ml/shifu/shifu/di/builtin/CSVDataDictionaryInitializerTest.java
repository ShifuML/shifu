package ml.shifu.shifu.di.builtin;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.xml.XmlMapper;
import ml.shifu.shifu.pmml.obj.DataDictionary;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.HashMap;

public class CSVDataDictionaryInitializerTest {

    @Test
    public void test() throws IOException {
        CSVDataDictionaryInitializer initializer = new CSVDataDictionaryInitializer();
        Map<String, Object> params = new HashMap<String, Object>();
        params.put("filePath", "src/test/resources/unittest/DataSet/iris/iris.csv");


        DataDictionary dict = initializer.init(params);

        ObjectMapper jsonMapper = new ObjectMapper();
        XmlMapper xmlMapper = new XmlMapper();

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(new File("../output.json"), dict);
        xmlMapper.writerWithDefaultPrettyPrinter().writeValue(new File("../output.xml"), dict);
        Assert.assertEquals(dict.getNumberOfFields(), 5);

    }

}
