import MainCFGScale from './MainCFGScale';
import MainHeight from './MainHeight';
import MainIterations from './MainIterations';
import MainSampler from './MainSampler';
import MainSteps from './MainSteps';
import MainWidth from './MainWidth';
import MainNano from './MainNano';

export const inputWidth = 'auto';

export default function MainSettings() {
  return (
    <div className="main-settings">
      <div className="main-settings-list">
        <div className="main-settings-row">
          <MainIterations />
          <MainSteps />
          <MainCFGScale />
        </div>
        <div className="main-settings-row">
          <MainWidth />
          <MainHeight />
          <MainSampler />
        </div>
        <div className="main-settings-row">
          <MainNano />
        </div>
      </div>
    </div>
  );
}
