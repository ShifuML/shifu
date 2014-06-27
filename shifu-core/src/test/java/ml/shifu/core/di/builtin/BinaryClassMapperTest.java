package ml.shifu.core.di.builtin;


import ml.shifu.core.di.builtin.derivedField.BinaryClassMapperCreator;
import ml.shifu.core.util.JSONUtils;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.*;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;

public class BinaryClassMapperTest {

    @Test
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

        pmml = PMMLUtils.loadPMML("test.xml");
        MapValues mapValues = (MapValues) pmml.getTransformationDictionary().getDerivedFields().get(0).getExpression();

        Assert.assertEquals(mapValues.getInlineTable().getRows().get(0).getContent().size(), 1);
    }

}
