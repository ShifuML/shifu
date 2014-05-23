package ml.shifu.shifu.di.builtin;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.dataformat.xml.XmlMapper;
import ml.shifu.shifu.pmml.obj.DataDictionary;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.HashMap;

public class TripletDataDictionaryTypeSetterTest {

    @Test
    public void test() throws IOException {
        TripletDataDictionaryTypeSetter typeSetter = new TripletDataDictionaryTypeSetter();

        XmlMapper xmlMapper = new XmlMapper();
        DataDictionary dict = xmlMapper.readValue(new File("src/test/resources/pmml/DataDictionaryInitialized.xml"), DataDictionary.class);

        Assert.assertNull(dict.getDataFields().get(0).getDataType());

        Map<String, Object> params = new HashMap<String, Object>();

        params.put("typeTripletFilePath", "src/test/resources/conf/DataTypeTripletIris.txt");

        typeSetter.setType(dict, params);

        Assert.assertEquals(dict.getDataFields().get(0).getDataType(), "double");
    }

}
