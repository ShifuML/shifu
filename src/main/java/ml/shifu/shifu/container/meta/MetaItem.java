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
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;

import org.apache.commons.collections.CollectionUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * ConfigMeta class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonInclude(Include.NON_NULL)
public class MetaItem implements Cloneable {

    private String name;

    private String type;

    private Object defval;

    private String directive;

    private String description;

    private Integer maxLength;

    private Integer minLength;

    private List<ValueOption> options;

    private String elementType;

    private List<MetaItem> element;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public Object getDefval() {
        return defval;
    }

    public void setDefval(Object defval) {
        this.defval = defval;
    }

    public String getDirective() {
        return directive;
    }

    public void setDirective(String directive) {
        this.directive = directive;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Integer getMaxLength() {
        return maxLength;
    }

    public void setMaxLength(Integer maxLength) {
        this.maxLength = maxLength;
    }

    public Integer getMinLength() {
        return minLength;
    }

    public void setMinLength(Integer minLength) {
        this.minLength = minLength;
    }

    public List<ValueOption> getOptions() {
        return options;
    }

    public void setOptions(List<ValueOption> options) {
        this.options = options;
    }

    public String getElementType() {
        return elementType;
    }

    public void setElementType(String elementType) {
        this.elementType = elementType;
    }

    public List<MetaItem> getElement() {
        return element;
    }

    public void setElement(List<MetaItem> element) {
        this.element = element;
    }

    @Override
    public MetaItem clone() {
        MetaItem copy = null;
        try {
            copy = (MetaItem)super.clone();
        } catch (CloneNotSupportedException e) {
            // This should never happen
            throw new InternalError(e.toString());
        }

        copy.setName(name);
        copy.setType(type);
        copy.setDefval(defval);
        copy.setMaxLength(maxLength);
        copy.setMinLength(minLength);
        copy.setOptions(options);
        copy.setElementType(elementType);

        if(CollectionUtils.isNotEmpty(element)) {
            List<MetaItem> elementList = new ArrayList<MetaItem>();
            for(MetaItem meta: element) {
                elementList.add(meta.clone());
            }

            copy.setElement(elementList);
        }

        return copy;
    }
}
