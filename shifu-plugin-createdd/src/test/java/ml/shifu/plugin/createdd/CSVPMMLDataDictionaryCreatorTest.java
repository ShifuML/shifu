/**
 * Copyright [2012-2014] eBay Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
