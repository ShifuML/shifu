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
package ml.shifu.shifu.meta;

import com.fasterxml.jackson.core.JsonGenerationException;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.shifu.container.meta.MetaGroup;
import ml.shifu.shifu.container.meta.MetaItem;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * ItemMetaGroupTest class
 */
public class ItemMetaGroupTest {

    private ObjectMapper jsonMapper;

    @BeforeClass
    public void setUp() {
        jsonMapper = new ObjectMapper();
    }

    @Test
    public void testOutput() throws JsonGenerationException, JsonMappingException, IOException {
        List<MetaGroup> groupList = new ArrayList<MetaGroup>();

        MetaGroup itemGrpA = new MetaGroup();
        itemGrpA.setGroup("basic");

        List<MetaItem> metaList = new ArrayList<MetaItem>();
        MetaItem meta = new MetaItem();
        meta.setName("author");
        meta.setType("text");
        meta.setDirective("input");
        meta.setMinLength(1);
        metaList.add(meta);
        itemGrpA.setMetaList(metaList);

        groupList.add(itemGrpA);

        MetaGroup itemGrpB = new MetaGroup();
        itemGrpB.setGroup("sourceData");

        List<MetaItem> metaListB = new ArrayList<MetaItem>();
        MetaItem metaB = new MetaItem();
        metaB.setName("dataPath");
        metaB.setType("text");
        metaB.setDirective("input");
        metaB.setMinLength(1);
        metaListB.add(metaB);
        itemGrpB.setMetaList(metaListB);

        groupList.add(itemGrpB);

        File file = new File("test-meta.json");
        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(file, groupList);

        MetaGroup[] ga = jsonMapper.readValue(file, MetaGroup[].class);
        Assert.assertEquals(groupList.size(), ga.length);

        file.deleteOnExit();
    }

    @Test
    public void testCloneMeta() {
        MetaGroup group = new MetaGroup();

        MetaGroup cloneObj = group.clone();
        Assert.assertNull(cloneObj.getGroup());

        group.setGroup("testGroup");
        List<MetaItem> itemList = new ArrayList<MetaItem>();
        MetaItem meta = new MetaItem();
        itemList.add(meta);
        itemList.add(meta.clone());
        itemList.add(meta.clone());
        group.setMetaList(itemList);

        cloneObj = group.clone();
        Assert.assertEquals("testGroup", cloneObj.getGroup());
        Assert.assertEquals(3, cloneObj.getMetaList().size());
    }

    @Test
    public void testReadMetaGroup() throws JsonParseException, JsonMappingException, IOException {
        File file = new File("src/main/resources/store/ModelConfigMeta.json");
        MetaGroup[] ga = jsonMapper.readValue(file, MetaGroup[].class);
        Assert.assertEquals(7, ga.length);
    }
}
