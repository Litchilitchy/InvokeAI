import { NANO_OPTIMS } from 'app/constants';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISelect from 'common/components/IAISelect';
import { setNano } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export default function MainNano() {
  const nano = useAppSelector((state: RootState) => state.generation.nano);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChangeNano = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setNano(String(e.target.value)));

  return (
    <IAISelect
      isDisabled={activeTabName === 'unifiedCanvas'}
      label="Nano Optimization"
      value={nano}
      flexGrow={1}
      onChange={handleChangeNano}
      validValues={NANO_OPTIMS}
      styleClass="main-settings-block"
    />
  );
}
