package ml.shifu.plugin.createdd;

import org.dmg.pmml.DataDictionary;
import org.testng.annotations.Test;
import java.io.File;

import ml.shifu.core.request.Request;
import ml.shifu.core.request.RequestDispatcher;
import ml.shifu.core.util.JSONUtils;
import ml.shifu.core.util.Params;

public class CSVPMMLDataDictionaryCreatorTest{

    @Test
    public void test1() throws Exception {
        /*
        RequestDispatcher
                .dispatch(JSONUtils.readValue(new File(
                        "src/test/resources/request/create.json"),
                        Request.class));
                        */
        Params param = new Params();
        param.put("csvDelimiter", "|");
        param.put("nameFileDelimiter", ",");
        param.put("pathCSV", "src/test/resources/data/wdbc/wdbc.header");
        param.put("columnNameFile", "src/test/resources/data/wdbc/columns.txt");
           
        CSVPMMLDataDictionaryCreator cpddc = new CSVPMMLDataDictionaryCreator();
        DataDictionary dd = cpddc.create(param);
        
        
        //now should verify the dd that gets created
    }


}
