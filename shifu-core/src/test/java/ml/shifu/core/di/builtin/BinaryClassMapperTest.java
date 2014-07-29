package ml.shifu.core.di.builtin;


import ml.shifu.core.di.builtin.derivedfield.BinaryClassMapperCreator;
import ml.shifu.core.util.JSONUtils;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.*;
import org.testng.Assert;

import java.io.File;

public class BinaryClassMapperTest {

    //@Test
    public void test() throws Exception {
        BinaryClassMapperCreator mapper = new BinaryClassMapperCreator();

        DataField dataField = new DataField();
        dataField.setName(new FieldName("target"));


        Params params = JSONUtils.readValue(new File("src/test/resources/request/BinaryClassMapperParams.json"), Params.class);

        DerivedField derivedField = mapper.create(dataField, null, params);

        PMML pmml = new PMML();
        TransformationDictionary dict = new TransformationDictionary();
        pmml.withTransformationDictionary(dict);
        dict.withDerivedFields(derivedField);
        PMMLUtils.savePMML(pmml, "test.xml");


        // something is wrong when reading the xml as inline table rows
        //pmml = PMMLUtils.loadPMML("test.xml");

        pmml = PMMLUtils.loadPMML("src/test/resources/pmml/ExampleMapValues.xml");
        MapValues mapValues = (MapValues) pmml.getTransformationDictionary().getDerivedFields().get(0).getExpression();

        Assert.assertEquals(mapValues.getInlineTable().getRows().get(0).getContent().size(), 1);
    }

}
