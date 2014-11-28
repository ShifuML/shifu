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

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * ModelVarSelectConf class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelVarSelectConf {

    // for force select or force remove
    private Boolean forceEnable = Boolean.TRUE;
    private String forceSelectColumnNameFile;
    private String forceRemoveColumnNameFile;

    // settings for variable selection
    private Boolean filterEnable = Boolean.TRUE;
    private Integer filterNum = Integer.valueOf(200);
    private String filterBy = "KS";

    // wrapper method for variable selection
    // don't open those options to user
    private Boolean wrapperEnabled = Boolean.FALSE;
    private Integer wrapperNum = Integer.valueOf(50);
    private Float wrapperRatio = Float.valueOf(0.05f);
    private String wrapperBy = "S";

    public Boolean getForceEnable() {
        return forceEnable;
    }

    public void setForceEnable(Boolean forceEnable) {
        this.forceEnable = forceEnable;
    }

    public String getForceSelectColumnNameFile() {
        return forceSelectColumnNameFile;
    }

    public void setForceSelectColumnNameFile(String forceSelectColumnConf) {
        this.forceSelectColumnNameFile = forceSelectColumnConf;
    }

    public String getForceRemoveColumnNameFile() {
        return forceRemoveColumnNameFile;
    }

    public void setForceRemoveColumnNameFile(String forceRemoveColumnConf) {
        this.forceRemoveColumnNameFile = forceRemoveColumnConf;
    }

    public Boolean getFilterEnable() {
        return filterEnable;
    }

    public void setFilterEnable(Boolean filterEnable) {
        this.filterEnable = filterEnable;
    }

    public Integer getFilterNum() {
        return filterNum;
    }

    public void setFilterNum(Integer filterNum) {
        this.filterNum = filterNum;
    }

    public String getFilterBy() {
        return filterBy;
    }

    public void setFilterBy(String filterBy) {
        this.filterBy = filterBy;
    }

    public Boolean getWrapperEnabled() {
        return wrapperEnabled;
    }

    public void setWrapperEnabled(Boolean wrapperEnabled) {
        this.wrapperEnabled = wrapperEnabled;
    }

    @JsonIgnore
    public Integer getWrapperNum() {
        return wrapperNum;
    }

    public void setWrapperNum(Integer wrapperNum) {
        this.wrapperNum = wrapperNum;
    }

    public String getWrapperBy() {
        return wrapperBy;
    }

    public void setWrapperBy(String wrapperBy) {
        this.wrapperBy = wrapperBy;
    }

    /**
     * @return the wrapperRatio
     */
    public Float getWrapperRatio() {
        return wrapperRatio;
    }

    /**
     * @param wrapperRatio the wrapperRatio to set
     */
    public void setWrapperRatio(Float wrapperRatio) {
        this.wrapperRatio = wrapperRatio;
    }

}
