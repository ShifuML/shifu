package ml.shifu.shifu.di.service;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.di.module.SimpleModule;
import ml.shifu.shifu.di.spi.DataDictionaryInitializer;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.DataDictionary;
import org.testng.Assert;
import org.testng.annotations.Test;

public class DataDictionaryServiceTest {

    @Test
    public void test() {
        /*
        Params params = new Params();
        params.put("filePath", "src/test/resources/conf/IrisFields.txt");


        SimpleModule module = new SimpleModule();
        module.set("DataDictionaryInitializer", "ml.shifu.shifu.di.builtin.dataDictionary.TripletDataDictionaryInitializer");
        Injector injector = Guice.createInjector(module);

        DataDictionaryService service = injector.getInstance(DataDictionaryService.class);
        DataDictionary dataDictionary = service.getDataDictionary(params);
        Assert.assertEquals((int)dataDictionary.getNumberOfFields(), 5);


        module.set("DataDictionaryInitializer", "ml.shifu.shifu.di.builtin.dataDictionary.ArffDataDictionaryInitializer");
        injector = Guice.createInjector(module);

        DataDictionaryService service2 = injector.getInstance(DataDictionaryService.class);
        DataDictionary dataDictionary2 = service2.getDataDictionary(params);
        Assert.assertEquals((int)dataDictionary2.getNumberOfFields(), 5);
        */
    }
}
