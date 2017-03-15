/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.dvarsel;

import java.util.List;

/**
 * Created by zhanhu on 2015/3/25.
 */
public class CandidateSeed {
    private int id;
    private List<Integer> columnIdList;

    public CandidateSeed(int id, List<Integer> columnIdList) {
        this.id = id;
        this.columnIdList = columnIdList;
    }

    public int getId() {
        return this.id;
    }

    public List<Integer> getColumnIdList() {
        return this.columnIdList;
    }

    public boolean sameAs(CandidateSeed worseSeed) {
        if (columnIdList.size() != worseSeed.getColumnIdList().size()) {
            return false;
        }
        for (Integer columnId : columnIdList) {
            if (!worseSeed.getColumnIdList().contains(columnId)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        return id;
    }

    @Override
    public boolean equals(Object obj) {
        if ( obj == null || !(obj instanceof CandidateSeed)) {
            return false;
        }

        CandidateSeed cs = (CandidateSeed) obj;
        if ( cs == this ) {
            return true;
        }

        return (id == cs.getId());
    }

    @Override
    public String toString() {
        return "CandidateSeed{"
                + "id=" + id
                + ", columnIdList=" + columnIdList
                + "}";
    }
}
