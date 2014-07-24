package ml.shifu.core.di.builtin;

import org.testng.annotations.Test;

import java.io.IOException;

public class TripletPMMLDataDictionaryCreatorTest {

    @Test
    public void test() throws IOException {
        /*
        TripletDataDictionaryInitializer initializer = new TripletDataDictionaryInitializer();

        Params params = new Params();

        params.put("filePath", "src/test/resources/conf/IrisFields.txt");

        DataDictionary dict = initializer.init(params);
        Assert.assertEquals((int)dict.getNumberOfFields(), 5);
        Assert.assertEquals(dict.getDataFields().get(0).getDataType().toString(), "DOUBLE");
        Assert.assertEquals(dict.getDataFields().get(0).getOptype().toString(), "CONTINUOUS");
        Assert.assertEquals(dict.getDataFields().get(4).getDataType().toString(), "STRING");
        Assert.assertEquals(dict.getDataFields().get(4).getOptype().toString(), "CATEGORICAL");

        ModelStats stats = new ModelStats();
        UnivariateStats univariateStats = new UnivariateStats();

        ContStats contStats = new ContStats();


        //stats.withUnivariateStats();  */

    }

}
