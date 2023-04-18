// EventManager.cpp
#include "GameEventManager.h"

TMap<FName, FOnGameEvent> UGameEventManager::EventMap;

void UGameEventManager::RegisterEvent(const FName& EventID, const FOnGameEvent& Callback)
{
    // ���¼���ӵ��¼�ӳ��
    EventMap.Add(EventID, Callback);
}

void UGameEventManager::UnregisterEvent(const FName& EventID)
{
    // �Ƴ��¼�
    EventMap.Remove(EventID);
}

void UGameEventManager::TriggerEvent(const FName& EventID, const FMessageData& Params)
{
    // �����¼�
    FOnGameEvent* EventDelegate = EventMap.Find(EventID);
    if (EventDelegate)
    {
        // �����¼�
        EventDelegate->ExecuteIfBound(Params);
    }
}
