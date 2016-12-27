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
package ml.shifu.shifu.container.obj;

import com.fasterxml.jackson.core.JsonGenerationException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.apache.commons.io.FileUtils;
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

    @BeforeClass
    public void setUp() {
        jsonMapper = new ObjectMapper();
    }

    @Test
    public void testSerial() throws JsonGenerationException, JsonMappingException, IOException {
        File file = new File("Columnfig.json");

        List<ColumnConfig> columnConfigList = new ArrayList<ColumnConfig>();

        ColumnConfig config = new ColumnConfig();
        config.setColumnName("TestColumn");
        config.setColumnStats(new ColumnStats());
        config.setColumnBinning(new ColumnBinning());

        columnConfigList.add(config);
        columnConfigList.add(config);

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(file, columnConfigList);

        List<ColumnConfig> ccList = Arrays.asList(jsonMapper.readValue(file, ColumnConfig[].class));
        
        Assert.assertEquals(2, ccList.size());
        Assert.assertEquals("TestColumn", ccList.get(0).getColumnName());

        FileUtils.deleteQuietly(file);
    }

    @Test
    public void testEmptyCategory() throws IOException {
        List<ColumnConfig> columnConfigList = new ArrayList<ColumnConfig>();
        ColumnConfig columnConfig = new ColumnConfig();
        
        columnConfig.setColumnNum(1);
        columnConfig.setColumnName("TestColumn");
        columnConfig.setColumnStats(new ColumnStats());
        columnConfig.setColumnBinning(new ColumnBinning());
        
        List<String> binCategoryList = new ArrayList<String>();
        
        binCategoryList.add("");
        binCategoryList.add("Hello\nWorld");
        binCategoryList.add("For\tTest");
        binCategoryList.add(";");
        binCategoryList.add(null);
        
        columnConfig.setBinCategory(binCategoryList);
        
        columnConfigList.add(columnConfig);
        
        File columnConfigFile = new File("ColumnConfig.json");
        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(columnConfigFile, columnConfigList);
        Assert.assertTrue(columnConfigFile.exists());
        
        FileUtils.deleteQuietly(columnConfigFile);
    }
}
