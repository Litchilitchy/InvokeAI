import {
  Flex,
  ListItem,
  Radio,
  RadioGroup,
  Text,
  UnorderedList,
  Tooltip,
} from '@chakra-ui/react';
import { convertToDiffusers } from 'app/socketio/actions';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

interface ModelConvertProps {
  model: string;
}

export default function ModelConvert(props: ModelConvertProps) {
  const { model } = props;

  const model_list = useAppSelector(
    (state: RootState) => state.system.model_list
  );

  const retrievedModel = model_list[model];

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const isConnected = useAppSelector(
    (state: RootState) => state.system.isConnected
  );

  const [saveLocation, setSaveLocation] = useState<string>('same');
  const [customSaveLocation, setCustomSaveLocation] = useState<string>('');

  useEffect(() => {
    setSaveLocation('same');
  }, [model]);

  const modelConvertCancelHandler = () => {
    setSaveLocation('same');
  };

  const modelConvertHandler = () => {
    const modelToConvert = {
      model_name: model,
      save_location: saveLocation,
      custom_location:
        saveLocation === 'custom' && customSaveLocation !== ''
          ? customSaveLocation
          : null,
    };
    dispatch(convertToDiffusers(modelToConvert));
  };

  return (
    <IAIAlertDialog
      title={`${t('modelmanager:convert')} ${model}`}
      acceptCallback={modelConvertHandler}
      cancelCallback={modelConvertCancelHandler}
      acceptButtonText={`${t('modelmanager:convert')}`}
      triggerComponent={
        <IAIButton
          size={'sm'}
          aria-label={t('modelmanager:convertToDiffusers')}
          isDisabled={
            retrievedModel.status === 'active' || isProcessing || !isConnected
          }
          className=" modal-close-btn"
          marginRight="2rem"
        >
          🧨 {t('modelmanager:convertToDiffusers')}
        </IAIButton>
      }
      motionPreset="slideInBottom"
    >
      <Flex flexDirection="column" rowGap={4}>
        <Text>{t('modelmanager:convertToDiffusersHelpText1')}</Text>
        <UnorderedList>
          <ListItem>{t('modelmanager:convertToDiffusersHelpText2')}</ListItem>
          <ListItem>{t('modelmanager:convertToDiffusersHelpText3')}</ListItem>
          <ListItem>{t('modelmanager:convertToDiffusersHelpText4')}</ListItem>
          <ListItem>{t('modelmanager:convertToDiffusersHelpText5')}</ListItem>
        </UnorderedList>
        <Text>{t('modelmanager:convertToDiffusersHelpText6')}</Text>
      </Flex>

      <Flex flexDir="column" gap={4}>
        <Flex marginTop="1rem" flexDir="column" gap={2}>
          <Text fontWeight="bold">
            {t('modelmanager:convertToDiffusersSaveLocation')}
          </Text>
          <RadioGroup value={saveLocation} onChange={(v) => setSaveLocation(v)}>
            <Flex gap={4}>
              <Radio value="same">
                <Tooltip label="Save converted model in the same folder">
                  {t('modelmanager:sameFolder')}
                </Tooltip>
              </Radio>

              <Radio value="root">
                <Tooltip label="Save converted model in the InvokeAI root folder">
                  {t('modelmanager:invokeRoot')}
                </Tooltip>
              </Radio>

              <Radio value="custom">
                <Tooltip label="Save converted model in a custom folder">
                  {t('modelmanager:custom')}
                </Tooltip>
              </Radio>
            </Flex>
          </RadioGroup>
        </Flex>

        {saveLocation === 'custom' && (
          <Flex flexDirection="column" rowGap={2}>
            <Text
              fontWeight="bold"
              fontSize="sm"
              color="var(--text-color-secondary)"
            >
              {t('modelmanager:customSaveLocation')}
            </Text>
            <IAIInput
              value={customSaveLocation}
              onChange={(e) => {
                if (e.target.value !== '')
                  setCustomSaveLocation(e.target.value);
              }}
              width="25rem"
            />
          </Flex>
        )}
      </Flex>
    </IAIAlertDialog>
  );
}
