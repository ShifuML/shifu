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
package ml.shifu.shifu.container.obj;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * ColumnConfigTest class
 */
public class ColumnConfigTest {

    private ObjectMapper jsonMapper;
    private File file;

    @BeforeClass
    public void setUp() {
        jsonMapper = new ObjectMapper();
        file = new File("ColumnConfig.json");
    }

    @Test
    public void testSaveLoad() throws IOException {

        if (file.exists()) {
            file.delete();
        }

        List<ColumnConfig> columnConfigList = new ArrayList<ColumnConfig>();

        ColumnConfig config = new ColumnConfig();
        config.setColumnName("TestColumn");

        config.getColumnNumStatsResult().getUserDefined().put("secondLargest", 1.0);

        columnConfigList.add(config);
        columnConfigList.add(config);

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(file, columnConfigList);

        List<ColumnConfig> ccList = Arrays.asList(jsonMapper.readValue(file, ColumnConfig[].class));

        Assert.assertEquals(2, ccList.size());
        Assert.assertEquals("TestColumn", ccList.get(0).getColumnName());
        Assert.assertEquals(ccList.get(0).getColumnNumStatsResult().getUserDefined().get("secondLargest"), 1.0);

        file.delete();
    }

}
