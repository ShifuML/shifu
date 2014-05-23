package ml.shifu.shifu.pmml.obj;

import com.fasterxml.jackson.dataformat.xml.XmlMapper;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;

public class DataDictionaryTest {

    @Test
    public void test() throws IOException {
        XmlMapper xmlMapper = new XmlMapper();
        DataDictionary dict = xmlMapper.readValue(new File("src/test/resources/pmml/DataDictionaryIris.xml"), DataDictionary.class);

        Assert.assertEquals(dict.getNumberOfFields(), 5);
        Assert.assertEquals(dict.getDataFields().get(0).getName(), "sepal_length");
        Assert.assertEquals(dict.getDataFields().get(0).getDataType(), "double");

        Assert.assertEquals(dict.getDataFields().get(4).getName(), "class");
        Assert.assertEquals(dict.getDataFields().get(4).getDataType(), "string");
    }
}
