import { render, screen, fireEvent } from '@testing-library/react';
import { IncidentDetailPanel } from '../AdvancedFireDashboard';

const baseIncident = {
  id: 'wfigs:bouquet',
  name: 'BOUQUET',
  county: 'Los Angeles',
  state: 'US-CA',
  containmentPct: 0,
  hasPrediction: true,
};

// The caption element itself. Matching on the word "perimeter" would also hit
// the Info tab's fine print, which mentions official perimeters.
const caption = (container) => container.querySelector('.incident-stat-source');

function renderPanel(incident) {
  return render(
    <IncidentDetailPanel
      incident={{ ...baseIncident, ...incident }}
      detail={null}
      activeTab="info"
      setActiveTab={() => {}}
      runningPrediction={false}
      onRunPrediction={() => {}}
      onClose={() => {}}
    />
  );
}

describe('incident acreage', () => {
  test('a measured perimeter size names its source and time', () => {
    // Bouquet: WFIGS overwrote its size with the 0.1-acre discovery figure
    // after FIRIS had mapped 1,048. The panel shows the measurement and says
    // where it came from, rather than silently disagreeing with the feed.
    renderPanel({
      acres: 1048,
      reportedAcres: 0.1,
      acresSource: { kind: 'perimeter', provider: 'FIRIS', asOf: '2026-10-04T04:09:00.000Z' },
    });

    expect(screen.getByText('1,048')).toBeInTheDocument();
    const shown = screen.getByText(/^FIRIS perimeter · /);
    expect(shown).toHaveClass('incident-stat-source');
    expect(shown.closest('div')).toHaveAttribute('title', 'WFIGS reports 0.1 acres');
  });

  test('a reported size carries no footnote', () => {
    const { container } = renderPanel({
      acres: 23448,
      reportedAcres: 23448,
      acresSource: { kind: 'reported', provider: 'WFIGS', asOf: '2026-10-04T03:30:00.000Z' },
    });

    expect(screen.getByText('23,448')).toBeInTheDocument();
    expect(caption(container)).toBeNull();
  });

  test('an incident from before this field existed still renders', () => {
    const { container } = renderPanel({ acres: 50 });
    expect(screen.getByText('50')).toBeInTheDocument();
    expect(caption(container)).toBeNull();
  });
});

describe('incident dates', () => {
  test('names the discovery date as such, apart from the feed record update', () => {
    renderPanel({ createdAt: '2026-10-03T22:19:00.000Z', updatedAt: '2026-10-04T13:28:00.000Z' });

    expect(screen.getByText('Discovered')).toBeInTheDocument();
    expect(screen.getByText('Feed record updated')).toBeInTheDocument();
    expect(screen.queryByText('Created')).not.toBeInTheDocument();
  });
});

describe('re-running a stale forecast', () => {
  test('the notice re-runs the forecast for the incident on screen', () => {
    const onRunPrediction = jest.fn();
    const incident = { ...baseIncident, hasPrediction: true };
    render(
      <IncidentDetailPanel
        incident={incident}
        detail={{ freshness: { lastDetectionAt: '2026-10-04T20:33:00.000Z', perimeterMappedAt: null, nextPasses: [] } }}
        activeTab="prediction"
        setActiveTab={() => {}}
        runningPrediction={false}
        onRunPrediction={onRunPrediction}
        newDetectionsSinceForecast
        onClose={() => {}}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: 'Re-run' }));

    expect(onRunPrediction).toHaveBeenCalledWith(incident);
  });
});

