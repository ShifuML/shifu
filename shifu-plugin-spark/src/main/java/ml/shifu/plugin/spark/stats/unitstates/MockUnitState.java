package ml.shifu.plugin.spark.stats.unitstates;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.UnivariateStats;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;

/**
 * A mock Unit State. Simply stores every object added to it in a list.
 * @author apalnitkar
 *
 */
public class MockUnitState implements UnitState {
	private static final long serialVersionUID = 1L;
	List<Object> objects;
	
	public MockUnitState() {
		objects= new ArrayList<Object>();
	}
	@Override
	public UnitState getNewBlank() {
		return new MockUnitState();
	}

	@Override
	public void merge(UnitState state) throws Exception {
		MockUnitState newState= (MockUnitState) state;
		this.objects.addAll(newState.getList());
	}

	@Override
	public void addData(Object value) {
		objects.add(value);
	}

	@Override
	public void populateUnivariateStats(UnivariateStats univariateStats,
			Params params) {
		return;
	}
	
	public List<Object> getList() {
		return this.objects;
	}
}
