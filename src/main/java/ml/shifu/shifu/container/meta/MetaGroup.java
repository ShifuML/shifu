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
package ml.shifu.shifu.container.meta;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import org.apache.commons.collections.CollectionUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * ItemMetaGroup class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class MetaGroup implements Cloneable {
    private String group;

    private List<MetaItem> metaList;

    public String getGroup() {
        return group;
    }

    public void setGroup(String group) {
        this.group = group;
    }

    public List<MetaItem> getMetaList() {
        return metaList;
    }

    public void setMetaList(List<MetaItem> metaList) {
        this.metaList = metaList;
    }

    @Override
    public MetaGroup clone() {
        MetaGroup metaGroup = null;
        try {
            metaGroup = (MetaGroup) super.clone();
        } catch (CloneNotSupportedException e) {
            // This should never happen
            throw new InternalError(e.toString());
        }
        

        // copy group
        metaGroup.setGroup(group);

        // copy meta list, if not null
        if(CollectionUtils.isNotEmpty(metaList)) {
            List<MetaItem> metas = new ArrayList<MetaItem>();
            for(MetaItem metaItem: metaList) {
                metas.add(metaItem.clone());
            }
            metaGroup.setMetaList(metas);
        }

        return metaGroup;
    }
}
