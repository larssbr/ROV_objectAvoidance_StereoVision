
Get a reference to the Vimba system object and list available system features.
C:\CV_projects\redBallObjectTracking>python referenceCam.py
System feature: CLTLIsPresent
System feature: DiscoveryCameraEvent
System feature: DiscoveryCameraIdent
System feature: DiscoveryInterfaceEvent
System feature: DiscoveryInterfaceIdent
System feature: Elapsed
System feature: FiWTLIsPresent
System feature: GeVDiscoveryAllAuto
System feature: GeVDiscoveryAllDuration
System feature: GeVDiscoveryAllOff
System feature: GeVDiscoveryAllOnce
System feature: GeVForceIPAddressGateway
System feature: GeVForceIPAddressIP
System feature: GeVForceIPAddressMAC
System feature: GeVForceIPAddressSend
System feature: GeVForceIPAddressSubnetMask
System feature: GeVTLIsPresent
System feature: UsbTLIsPresent



# trying to run testCam.py gives

C:\CV_projects\redBallObjectTracking>python testCam.py
Camera ID: DEV_000F31026082
Camera ID: DEV_000F31021DCD
Traceback (most recent call last):
  File "testCam.py", line 20, in <module>
    camera0.openCamera()
  File "C:\Anaconda2\lib\site-packages\pymba-0.1-py2.7.egg\pymba\vimbacamera.py", line 74, in openCamera
    raise VimbaException(errorCode)
pymba.vimbaexception.VimbaException: Operation is invalid with the current access mode.


## comments:
    we get two camera ID. --> GOOD
    
    we get error code -6 
    # we don't know what this means yet
    
    
    --> then i configured ipv4 in network settings to the address: 192.168.1.111
    
# running testCam.py again gives:

C:\CV_projects\redBallObjectTracking>python testCam.py
Camera ID: DEV_000F31026082
Camera ID: DEV_000F31021DCD
Camera feature: AcquisitionAbort
Camera feature: AcquisitionFrameCount
Camera feature: AcquisitionFrameRateAbs
Camera feature: AcquisitionFrameRateLimit
Camera feature: AcquisitionMode
Camera feature: AcquisitionStart
Camera feature: AcquisitionStop
Camera feature: BalanceRatioAbs
Camera feature: BalanceRatioSelector
Camera feature: BalanceWhiteAuto
Camera feature: BalanceWhiteAutoAdjustTol
Camera feature: BalanceWhiteAutoRate
Camera feature: BandwidthControlMode
Camera feature: BinningHorizontal
Camera feature: BinningVertical
Camera feature: BlackLevelSelector
Camera feature: ChunkModeActive
Camera feature: DSPSubregionBottom
Camera feature: DSPSubregionLeft
Camera feature: DSPSubregionRight
Camera feature: DSPSubregionTop
Camera feature: DeviceFirmwareVersion
Camera feature: DeviceID
Camera feature: DeviceModelName
Camera feature: DevicePartNumber
Camera feature: DeviceScanType
Camera feature: DeviceUserID
Camera feature: DeviceVendorName
Camera feature: EventAcquisitionEnd
Camera feature: EventAcquisitionEndFrameID
Camera feature: EventAcquisitionEndTimestamp
Camera feature: EventAcquisitionRecordTrigger
Camera feature: EventAcquisitionRecordTriggerFrameID
Camera feature: EventAcquisitionRecordTriggerTimestamp
Camera feature: EventAcquisitionStart
Camera feature: EventAcquisitionStartFrameID
Camera feature: EventAcquisitionStartTimestamp
Camera feature: EventError
Camera feature: EventErrorFrameID
Camera feature: EventErrorTimestamp
Camera feature: EventExposureEnd
Camera feature: EventExposureEndFrameID
Camera feature: EventExposureEndTimestamp
Camera feature: EventFrameTrigger
Camera feature: EventFrameTriggerFrameID
Camera feature: EventFrameTriggerReady
Camera feature: EventFrameTriggerReadyFrameID
Camera feature: EventFrameTriggerReadyTimestamp
Camera feature: EventFrameTriggerTimestamp
Camera feature: EventLine1FallingEdge
Camera feature: EventLine1FallingEdgeFrameID
Camera feature: EventLine1FallingEdgeTimestamp
Camera feature: EventLine1RisingEdge
Camera feature: EventLine1RisingEdgeFrameID
Camera feature: EventLine1RisingEdgeTimestamp
Camera feature: EventLine2FallingEdge
Camera feature: EventLine2FallingEdgeFrameID
Camera feature: EventLine2FallingEdgeTimestamp
Camera feature: EventLine2RisingEdge
Camera feature: EventLine2RisingEdgeFrameID
Camera feature: EventLine2RisingEdgeTimestamp
Camera feature: EventLine3FallingEdge
Camera feature: EventLine3FallingEdgeFrameID
Camera feature: EventLine3FallingEdgeTimestamp
Camera feature: EventLine3RisingEdge
Camera feature: EventLine3RisingEdgeFrameID
Camera feature: EventLine3RisingEdgeTimestamp
Camera feature: EventLine4FallingEdge
Camera feature: EventLine4FallingEdgeFrameID
Camera feature: EventLine4FallingEdgeTimestamp
Camera feature: EventLine4RisingEdge
Camera feature: EventLine4RisingEdgeFrameID
Camera feature: EventLine4RisingEdgeTimestamp
Camera feature: EventNotification
Camera feature: EventOverflow
Camera feature: EventOverflowFrameID
Camera feature: EventOverflowTimestamp
Camera feature: EventPtpSyncLocked
Camera feature: EventPtpSyncLockedFrameID
Camera feature: EventPtpSyncLockedTimestamp
Camera feature: EventPtpSyncLost
Camera feature: EventPtpSyncLostFrameID
Camera feature: EventPtpSyncLostTimestamp
Camera feature: EventSelector
Camera feature: EventsEnable1
Camera feature: ExposureAuto
Camera feature: ExposureAutoAdjustTol
Camera feature: ExposureAutoAlg
Camera feature: ExposureAutoMax
Camera feature: ExposureAutoMin
Camera feature: ExposureAutoOutliers
Camera feature: ExposureAutoRate
Camera feature: ExposureAutoTarget
Camera feature: ExposureMode
Camera feature: ExposureTimeAbs
Camera feature: FirmwareVerBuild
Camera feature: FirmwareVerMajor
Camera feature: FirmwareVerMinor
Camera feature: GVCPCmdRetries
Camera feature: GVCPCmdTimeout
Camera feature: GVSPAdjustPacketSize
Camera feature: GVSPBurstSize
Camera feature: GVSPDriver
Camera feature: GVSPFilterVersion
Camera feature: GVSPHostReceiveBuffers
Camera feature: GVSPMaxLookBack
Camera feature: GVSPMaxRequests
Camera feature: GVSPMaxWaitSize
Camera feature: GVSPMissingSize
Camera feature: GVSPPacketSize
Camera feature: GVSPTiltingSize
Camera feature: GVSPTimeout
Camera feature: GainAuto
Camera feature: GainAutoAdjustTol
Camera feature: GainAutoMax
Camera feature: GainAutoMin
Camera feature: GainAutoOutliers
Camera feature: GainAutoRate
Camera feature: GainAutoTarget
Camera feature: GainRaw
Camera feature: GainSelector
Camera feature: GevCurrentDefaultGateway
Camera feature: GevCurrentIPAddress
Camera feature: GevCurrentSubnetMask
Camera feature: GevDeviceMACAddress
Camera feature: GevHeartbeatInterval
Camera feature: GevHeartbeatTimeout
Camera feature: GevIPConfigurationMode
Camera feature: GevPersistentDefaultGateway
Camera feature: GevPersistentIPAddress
Camera feature: GevPersistentSubnetMask
Camera feature: GevSCPSPacketSize
Camera feature: GevTimestampControlLatch
Camera feature: GevTimestampControlReset
Camera feature: GevTimestampTickFrequency
Camera feature: GevTimestampValue
Camera feature: Height
Camera feature: HeightMax
Camera feature: ImageSize
Camera feature: IrisAutoTarget
Camera feature: IrisMode
Camera feature: IrisVideoLevel
Camera feature: IrisVideoLevelMax
Camera feature: IrisVideoLevelMin
Camera feature: MulticastEnable
Camera feature: MulticastIPAddress
Camera feature: NonImagePayloadSize
Camera feature: OffsetX
Camera feature: OffsetY
Camera feature: PayloadSize
Camera feature: PixelFormat
Camera feature: PtpAcquisitionGateTime
Camera feature: PtpMode
Camera feature: PtpStatus
Camera feature: RecorderPreEventCount
Camera feature: SensorBits
Camera feature: SensorHeight
Camera feature: SensorType
Camera feature: SensorWidth
Camera feature: StatFrameDelivered
Camera feature: StatFrameDropped
Camera feature: StatFrameRate
Camera feature: StatFrameRescued
Camera feature: StatFrameShoved
Camera feature: StatFrameUnderrun
Camera feature: StatLocalRate
Camera feature: StatPacketErrors
Camera feature: StatPacketMissed
Camera feature: StatPacketReceived
Camera feature: StatPacketRequested
Camera feature: StatPacketResent
Camera feature: StatTimeElapsed
Camera feature: StreamAnnounceBufferMinimum
Camera feature: StreamAnnouncedBufferCount
Camera feature: StreamBufferHandlingMode
Camera feature: StreamBytesPerSecond
Camera feature: StreamFrameRateConstrain
Camera feature: StreamHoldCapacity
Camera feature: StreamHoldEnable
Camera feature: StreamID
Camera feature: StreamType
Camera feature: StrobeDelay
Camera feature: StrobeDuration
Camera feature: StrobeDurationMode
Camera feature: StrobeSource
Camera feature: SyncInGlitchFilter
Camera feature: SyncInLevels
Camera feature: SyncInSelector
Camera feature: SyncOutLevels
Camera feature: SyncOutPolarity
Camera feature: SyncOutSelector
Camera feature: SyncOutSource
Camera feature: TriggerActivation
Camera feature: TriggerDelayAbs
Camera feature: TriggerMode
Camera feature: TriggerOverlap
Camera feature: TriggerSelector
Camera feature: TriggerSoftware
Camera feature: TriggerSource
Camera feature: UserSetDefaultSelector
Camera feature: UserSetLoad
Camera feature: UserSetSave
Camera feature: UserSetSelector
Camera feature: Width
Camera feature: WidthMax
Continuous


