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
package ml.shifu.shifu.container;

/**
 * variable store, the variable would be store in a json file which could help to know the variable type(C/N)
 */
public class VariableStoreObject {
    // Unique Name
    private String varName;

    // Numerical, Categorical, Identifier
    private String varType;

    // TSC / MEP / NTSC, etc
    private String varGroupName;

    // The GOTO Person
    private String varCreator;

    // If it is a derived variable
    private String flagDerived;

    // e.g. derived_max_foo_bar
    private String derivedSchema;

    public String getVarName() {
        return varName;
    }

    public void setVarName(String varName) {
        this.varName = varName;
    }

    public String getVarType() {
        return varType;
    }

    public void setVarType(String varType) {
        this.varType = varType;
    }

    public String getVarGroupName() {
        return varGroupName;
    }

    public void setVarGroupName(String varGroupName) {
        this.varGroupName = varGroupName;
    }

    public String getVarCreator() {
        return varCreator;
    }

    public void setVarCreator(String varCreator) {
        this.varCreator = varCreator;
    }

    public String getFlagDerived() {
        return flagDerived;
    }

    public void setFlagDerived(String flagDerived) {
        this.flagDerived = flagDerived;
    }

    public String getDerivedSchema() {
        return derivedSchema;
    }

    public void setDerivedSchema(String derivedSchema) {
        this.derivedSchema = derivedSchema;
    }

}
