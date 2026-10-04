import { render, screen } from '@testing-library/react';
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
