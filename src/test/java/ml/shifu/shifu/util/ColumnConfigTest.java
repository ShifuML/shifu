/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.util;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.shifu.container.obj.ColumnConfig;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class ColumnConfigTest {

    //@Test
    public void testSaveColumnConfig() throws JsonParseException, JsonMappingException, IOException {

        ColumnConfig[] configs = new ColumnConfig[5];
        for (int i = 0; i < 5; i++) {
            configs[i] = new ColumnConfig();
            configs[i].setColumnName("column" + i);
        }


        ObjectMapper mapper = new ObjectMapper();
        mapper.writeValue(new File("src/test/resources/reason_data/test2.json"), configs);

    }

    //@Test
    public void testLoadColumnConfig() throws JsonParseException, JsonMappingException, IOException {
        ObjectMapper mapper = new ObjectMapper();
        ColumnConfig[] configArray = mapper.readValue(new File("src/test/resources/reason_data/test2.json"),
                ColumnConfig[].class);
        List<ColumnConfig> configs = Arrays.asList(configArray);
        ColumnConfig config = configs.get(1);
        System.out.println(config.getColumnName());
    }

}
